import { Module } from '@nestjs/common';
import { TranslateModule } from './translate/translate.module';
import { ConfigModule } from '@nestjs/config';
import { TypeOrmModule } from '@nestjs/typeorm';
import { Translate } from './entities/translate.entity';

@Module({
  imports: [
    ConfigModule.forRoot(),
    TypeOrmModule.forRoot({
      type: 'postgres',
      host: process.env.DB_HOST,
      port: parseInt(process.env.DB_PORT),
      username: process.env.DB_USER,
      password: process.env.DB_PASS,
      database: process.env.DB_NAME,
      entities: [Translate],
      synchronize: true,
    }),
    TranslateModule,
  ],
  controllers: [],
  providers: [],
})
export class AppModule {}
